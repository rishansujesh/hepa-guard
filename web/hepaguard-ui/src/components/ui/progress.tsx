import * as React from "react";

import { cn } from "@/lib/utils";

interface ProgressProps extends React.HTMLAttributes<HTMLDivElement> {
  value: number;
}

export function Progress({ value, className, ...props }: ProgressProps) {
  const safeValue = Math.min(100, Math.max(0, value));
  return (
    <div
      className={cn("h-2 w-full rounded-full bg-slate-200", className)}
      {...props}
    >
      <div
        className="h-full rounded-full bg-blue-600 transition-all"
        style={{ width: `${safeValue}%` }}
      />
    </div>
  );
}
